; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 515249

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 - child
    bread153 - bread-portion
    content45 - content-portion
    tray480 tray280 tray16 tray27 - tray
    table328 table277 table464 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray480 kitchen)
     (at tray280 kitchen)
     (at tray16 kitchen)
     (at tray27 kitchen)
     (at_kitchen_bread bread153)
     (at_kitchen_content content45)
     (not_allergic_gluten child316)
     (waiting child316 table277)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child316)
    )
  )
)

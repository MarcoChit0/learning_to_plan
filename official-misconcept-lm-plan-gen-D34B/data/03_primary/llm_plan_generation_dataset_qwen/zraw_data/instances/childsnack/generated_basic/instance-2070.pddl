; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 488656

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child483 - child
    bread324 - bread-portion
    content143 - content-portion
    tray356 tray407 tray149 tray329 - tray
    table378 table114 table437 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at tray407 kitchen)
     (at tray149 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_content content143)
     (not_allergic_gluten child483)
     (waiting child483 table114)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child483)
    )
  )
)

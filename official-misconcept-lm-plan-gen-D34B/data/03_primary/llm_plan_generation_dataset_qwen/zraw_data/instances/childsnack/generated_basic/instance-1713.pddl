; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 397554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child208 - child
    bread475 - bread-portion
    content143 - content-portion
    tray213 tray428 tray432 - tray
    table217 table469 table397 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray213 kitchen)
     (at tray428 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_content content143)
     (not_allergic_gluten child208)
     (waiting child208 table397)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child208)
    )
  )
)

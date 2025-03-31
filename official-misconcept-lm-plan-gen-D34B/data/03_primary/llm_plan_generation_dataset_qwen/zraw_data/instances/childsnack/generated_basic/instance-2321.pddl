; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 49921

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child483 - child
    bread269 - bread-portion
    content48 - content-portion
    tray202 - tray
    table61 table221 table3 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray202 kitchen)
     (at_kitchen_bread bread269)
     (at_kitchen_content content48)
     (not_allergic_gluten child483)
     (waiting child483 table61)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child483)
    )
  )
)

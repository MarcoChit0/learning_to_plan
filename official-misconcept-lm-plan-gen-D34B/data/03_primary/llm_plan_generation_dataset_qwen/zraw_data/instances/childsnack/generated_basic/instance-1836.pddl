; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 600194

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child40 child436 - child
    bread494 bread265 - bread-portion
    content129 content477 - content-portion
    tray16 - tray
    table497 table319 table161 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray16 kitchen)
     (at_kitchen_bread bread494)
     (at_kitchen_bread bread265)
     (at_kitchen_content content129)
     (at_kitchen_content content477)
     (no_gluten_bread bread494)
     (no_gluten_content content129)
     (allergic_gluten child436)
     (not_allergic_gluten child40)
     (waiting child40 table497)
     (waiting child436 table497)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child40)
     (served child436)
    )
  )
)

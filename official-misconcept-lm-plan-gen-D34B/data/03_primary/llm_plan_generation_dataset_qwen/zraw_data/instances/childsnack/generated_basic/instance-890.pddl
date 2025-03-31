; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 661692

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child260 child167 child176 - child
    bread324 bread226 bread282 - bread-portion
    content327 content163 content265 - content-portion
    tray113 - tray
    table187 table301 table492 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray113 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread282)
     (at_kitchen_content content327)
     (at_kitchen_content content163)
     (at_kitchen_content content265)
     (not_allergic_gluten child176)
     (not_allergic_gluten child260)
     (not_allergic_gluten child167)
     (waiting child260 table492)
     (waiting child167 table187)
     (waiting child176 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child260)
     (served child167)
     (served child176)
    )
  )
)

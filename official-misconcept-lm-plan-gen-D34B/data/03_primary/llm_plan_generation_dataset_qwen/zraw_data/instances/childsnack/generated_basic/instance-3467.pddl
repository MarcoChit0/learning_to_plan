; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 740355

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 child49 - child
    bread266 bread11 - bread-portion
    content44 content182 - content-portion
    tray212 tray51 - tray
    table270 table249 table31 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at tray51 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_bread bread11)
     (at_kitchen_content content44)
     (at_kitchen_content content182)
     (no_gluten_bread bread266)
     (no_gluten_content content44)
     (allergic_gluten child68)
     (not_allergic_gluten child49)
     (waiting child68 table249)
     (waiting child49 table249)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child68)
     (served child49)
    )
  )
)

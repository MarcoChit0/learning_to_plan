; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 711586

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child404 child136 - child
    bread320 bread226 - bread-portion
    content106 content208 - content-portion
    tray302 tray344 - tray
    table208 table258 table287 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray344 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_bread bread226)
     (at_kitchen_content content106)
     (at_kitchen_content content208)
     (no_gluten_bread bread226)
     (no_gluten_content content106)
     (allergic_gluten child404)
     (not_allergic_gluten child136)
     (waiting child404 table287)
     (waiting child136 table258)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child404)
     (served child136)
    )
  )
)

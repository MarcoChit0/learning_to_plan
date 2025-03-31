; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 373472

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child161 child138 child311 - child
    bread245 bread103 bread371 - bread-portion
    content139 content248 content380 - content-portion
    tray182 - tray
    table376 table404 table315 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at_kitchen_bread bread245)
     (at_kitchen_bread bread103)
     (at_kitchen_bread bread371)
     (at_kitchen_content content139)
     (at_kitchen_content content248)
     (at_kitchen_content content380)
     (not_allergic_gluten child161)
     (not_allergic_gluten child138)
     (not_allergic_gluten child311)
     (waiting child161 table404)
     (waiting child138 table376)
     (waiting child311 table376)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child161)
     (served child138)
     (served child311)
    )
  )
)

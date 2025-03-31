; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 716430

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child403 child272 - child
    bread344 bread156 - bread-portion
    content259 content34 - content-portion
    tray223 tray349 tray361 - tray
    table315 table288 table290 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray223 kitchen)
     (at tray349 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread344)
     (at_kitchen_bread bread156)
     (at_kitchen_content content259)
     (at_kitchen_content content34)
     (not_allergic_gluten child403)
     (not_allergic_gluten child272)
     (waiting child403 table315)
     (waiting child272 table315)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child403)
     (served child272)
    )
  )
)

; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 488044

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 child380 - child
    bread498 bread212 - bread-portion
    content334 content135 - content-portion
    tray338 tray119 - tray
    table279 table389 table199 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at tray119 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_bread bread212)
     (at_kitchen_content content334)
     (at_kitchen_content content135)
     (not_allergic_gluten child380)
     (not_allergic_gluten child4)
     (waiting child4 table279)
     (waiting child380 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child4)
     (served child380)
    )
  )
)

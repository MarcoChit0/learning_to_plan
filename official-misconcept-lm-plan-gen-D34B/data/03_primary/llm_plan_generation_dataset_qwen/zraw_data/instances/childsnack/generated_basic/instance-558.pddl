; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 707742

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child287 child323 - child
    bread478 bread490 - bread-portion
    content53 content208 - content-portion
    tray373 - tray
    table359 table389 table251 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at_kitchen_bread bread478)
     (at_kitchen_bread bread490)
     (at_kitchen_content content53)
     (at_kitchen_content content208)
     (no_gluten_bread bread490)
     (no_gluten_content content208)
     (allergic_gluten child323)
     (not_allergic_gluten child287)
     (waiting child287 table359)
     (waiting child323 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child287)
     (served child323)
    )
  )
)

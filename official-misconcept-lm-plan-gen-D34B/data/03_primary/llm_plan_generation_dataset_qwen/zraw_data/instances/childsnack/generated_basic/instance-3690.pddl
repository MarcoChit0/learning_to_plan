; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 912926

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child101 child323 - child
    bread243 bread175 - bread-portion
    content334 content461 - content-portion
    tray371 tray345 - tray
    table213 table204 table45 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray345 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread175)
     (at_kitchen_content content334)
     (at_kitchen_content content461)
     (no_gluten_bread bread175)
     (no_gluten_content content334)
     (allergic_gluten child323)
     (not_allergic_gluten child101)
     (waiting child101 table204)
     (waiting child323 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child101)
     (served child323)
    )
  )
)

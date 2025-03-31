; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 783648

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child183 child97 - child
    bread201 bread54 - bread-portion
    content493 content269 - content-portion
    tray430 tray212 - tray
    table490 table300 table383 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray430 kitchen)
     (at tray212 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_bread bread54)
     (at_kitchen_content content493)
     (at_kitchen_content content269)
     (no_gluten_bread bread201)
     (no_gluten_content content269)
     (allergic_gluten child97)
     (not_allergic_gluten child183)
     (waiting child183 table490)
     (waiting child97 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child183)
     (served child97)
    )
  )
)

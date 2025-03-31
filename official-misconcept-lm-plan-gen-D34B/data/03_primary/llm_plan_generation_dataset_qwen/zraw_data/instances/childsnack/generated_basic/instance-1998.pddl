; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 498544

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 child183 - child
    bread355 bread226 - bread-portion
    content208 content268 - content-portion
    tray208 - tray
    table434 table352 table53 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray208 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_bread bread226)
     (at_kitchen_content content208)
     (at_kitchen_content content268)
     (no_gluten_bread bread355)
     (no_gluten_content content208)
     (allergic_gluten child183)
     (not_allergic_gluten child131)
     (waiting child131 table352)
     (waiting child183 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child131)
     (served child183)
    )
  )
)

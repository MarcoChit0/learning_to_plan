; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 914184

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child197 child3 - child
    bread208 bread291 - bread-portion
    content441 content165 - content-portion
    tray300 - tray
    table387 table410 table58 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray300 kitchen)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread291)
     (at_kitchen_content content441)
     (at_kitchen_content content165)
     (no_gluten_bread bread208)
     (no_gluten_content content165)
     (allergic_gluten child3)
     (not_allergic_gluten child197)
     (waiting child197 table410)
     (waiting child3 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child197)
     (served child3)
    )
  )
)

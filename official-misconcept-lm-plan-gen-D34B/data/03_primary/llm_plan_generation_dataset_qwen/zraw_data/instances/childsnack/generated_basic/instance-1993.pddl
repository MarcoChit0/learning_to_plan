; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 874311

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 child245 - child
    bread235 bread178 - bread-portion
    content476 content312 - content-portion
    tray363 - tray
    table105 table67 table46 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at_kitchen_bread bread235)
     (at_kitchen_bread bread178)
     (at_kitchen_content content476)
     (at_kitchen_content content312)
     (no_gluten_bread bread178)
     (no_gluten_content content312)
     (allergic_gluten child463)
     (not_allergic_gluten child245)
     (waiting child463 table46)
     (waiting child245 table46)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child463)
     (served child245)
    )
  )
)

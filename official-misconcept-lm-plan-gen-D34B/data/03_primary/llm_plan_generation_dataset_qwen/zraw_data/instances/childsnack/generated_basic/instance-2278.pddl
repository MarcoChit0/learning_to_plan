; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 699632

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child173 - child
    bread10 bread394 - bread-portion
    content241 content371 - content-portion
    tray140 - tray
    table498 table116 table499 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray140 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread394)
     (at_kitchen_content content241)
     (at_kitchen_content content371)
     (no_gluten_bread bread10)
     (no_gluten_content content241)
     (allergic_gluten child173)
     (not_allergic_gluten child225)
     (waiting child225 table499)
     (waiting child173 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child225)
     (served child173)
    )
  )
)

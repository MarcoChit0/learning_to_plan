; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 185704

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 - child
    bread396 - bread-portion
    content108 - content-portion
    tray167 - tray
    table364 table123 table5 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at_kitchen_bread bread396)
     (at_kitchen_content content108)
     (not_allergic_gluten child455)
     (waiting child455 table123)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child455)
    )
  )
)

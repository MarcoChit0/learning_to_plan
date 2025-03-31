; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 845726

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 child52 - child
    bread239 bread281 - bread-portion
    content422 content67 - content-portion
    tray439 - tray
    table9 table5 table363 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread281)
     (at_kitchen_content content422)
     (at_kitchen_content content67)
     (no_gluten_bread bread281)
     (no_gluten_content content422)
     (allergic_gluten child52)
     (not_allergic_gluten child396)
     (waiting child396 table5)
     (waiting child52 table9)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child396)
     (served child52)
    )
  )
)

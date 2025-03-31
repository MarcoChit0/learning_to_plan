; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 462744

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child404 child77 - child
    bread67 bread99 - bread-portion
    content70 content279 - content-portion
    tray40 tray469 - tray
    table129 table65 table252 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread67)
     (at_kitchen_bread bread99)
     (at_kitchen_content content70)
     (at_kitchen_content content279)
     (no_gluten_bread bread67)
     (no_gluten_content content279)
     (allergic_gluten child77)
     (not_allergic_gluten child404)
     (waiting child404 table65)
     (waiting child77 table129)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child404)
     (served child77)
    )
  )
)

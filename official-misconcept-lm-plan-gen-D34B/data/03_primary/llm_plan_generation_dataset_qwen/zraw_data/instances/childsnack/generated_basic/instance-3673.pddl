; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 114559

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child6 - child
    bread488 bread301 - bread-portion
    content220 content416 - content-portion
    tray210 tray435 - tray
    table446 table61 table342 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray210 kitchen)
     (at tray435 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_bread bread301)
     (at_kitchen_content content220)
     (at_kitchen_content content416)
     (no_gluten_bread bread301)
     (no_gluten_content content416)
     (allergic_gluten child6)
     (not_allergic_gluten child62)
     (waiting child62 table342)
     (waiting child6 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child62)
     (served child6)
    )
  )
)

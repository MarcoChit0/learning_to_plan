; child-snack task with 3 children and 0.74 gluten factor 
; constant factor of 1.3
; random seed: 105587

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child426 child487 child179 - child
    bread190 bread301 bread84 - bread-portion
    content472 content172 content334 - content-portion
    tray306 - tray
    table492 table218 table145 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_bread bread301)
     (at_kitchen_bread bread84)
     (at_kitchen_content content472)
     (at_kitchen_content content172)
     (at_kitchen_content content334)
     (no_gluten_bread bread190)
     (no_gluten_bread bread301)
     (no_gluten_content content472)
     (no_gluten_content content172)
     (allergic_gluten child487)
     (allergic_gluten child426)
     (not_allergic_gluten child179)
     (waiting child426 table492)
     (waiting child487 table492)
     (waiting child179 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child426)
     (served child487)
     (served child179)
    )
  )
)

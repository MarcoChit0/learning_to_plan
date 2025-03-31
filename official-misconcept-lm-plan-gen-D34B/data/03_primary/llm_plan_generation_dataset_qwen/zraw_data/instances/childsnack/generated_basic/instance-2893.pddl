; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 939684

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 child97 - child
    bread421 bread239 - bread-portion
    content222 content349 - content-portion
    tray205 tray37 tray243 - tray
    table439 table216 table345 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray205 kitchen)
     (at tray37 kitchen)
     (at tray243 kitchen)
     (at_kitchen_bread bread421)
     (at_kitchen_bread bread239)
     (at_kitchen_content content222)
     (at_kitchen_content content349)
     (no_gluten_bread bread239)
     (no_gluten_content content349)
     (allergic_gluten child97)
     (not_allergic_gluten child398)
     (waiting child398 table439)
     (waiting child97 table345)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child398)
     (served child97)
    )
  )
)

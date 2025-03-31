; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 745413

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child263 child241 - child
    bread48 bread403 - bread-portion
    content327 content332 - content-portion
    tray348 tray120 - tray
    table92 table261 table175 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray348 kitchen)
     (at tray120 kitchen)
     (at_kitchen_bread bread48)
     (at_kitchen_bread bread403)
     (at_kitchen_content content327)
     (at_kitchen_content content332)
     (not_allergic_gluten child241)
     (not_allergic_gluten child263)
     (waiting child263 table261)
     (waiting child241 table92)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child263)
     (served child241)
    )
  )
)

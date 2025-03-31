; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 68378

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child423 - child
    bread315 bread317 - bread-portion
    content115 content42 - content-portion
    tray373 tray363 - tray
    table337 table241 table221 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at tray363 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_bread bread317)
     (at_kitchen_content content115)
     (at_kitchen_content content42)
     (no_gluten_bread bread315)
     (no_gluten_content content42)
     (allergic_gluten child279)
     (not_allergic_gluten child423)
     (waiting child279 table241)
     (waiting child423 table241)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child279)
     (served child423)
    )
  )
)

; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 275964

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child150 child426 - child
    bread206 bread231 - bread-portion
    content7 content257 - content-portion
    tray351 - tray
    table203 table273 table313 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at_kitchen_bread bread206)
     (at_kitchen_bread bread231)
     (at_kitchen_content content7)
     (at_kitchen_content content257)
     (no_gluten_bread bread231)
     (no_gluten_content content257)
     (allergic_gluten child426)
     (not_allergic_gluten child150)
     (waiting child150 table273)
     (waiting child426 table313)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child150)
     (served child426)
    )
  )
)

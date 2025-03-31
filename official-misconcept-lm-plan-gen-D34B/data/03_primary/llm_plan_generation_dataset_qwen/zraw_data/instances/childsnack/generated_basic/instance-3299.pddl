; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 198092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 child314 - child
    bread229 bread272 - bread-portion
    content200 content99 - content-portion
    tray439 tray161 tray73 tray475 - tray
    table458 table263 table141 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at tray161 kitchen)
     (at tray73 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_bread bread272)
     (at_kitchen_content content200)
     (at_kitchen_content content99)
     (no_gluten_bread bread272)
     (no_gluten_content content99)
     (allergic_gluten child314)
     (not_allergic_gluten child407)
     (waiting child407 table141)
     (waiting child314 table263)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child407)
     (served child314)
    )
  )
)

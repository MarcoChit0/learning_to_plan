; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 825703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 - child
    bread381 - bread-portion
    content229 - content-portion
    tray478 - tray
    table420 table276 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_content content229)
     (not_allergic_gluten child314)
     (waiting child314 table276)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child314)
    )
  )
)

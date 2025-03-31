; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 767001

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 child400 - child
    bread170 bread317 - bread-portion
    content142 content221 - content-portion
    tray359 tray287 tray72 - tray
    table188 table165 table453 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray359 kitchen)
     (at tray287 kitchen)
     (at tray72 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_bread bread317)
     (at_kitchen_content content142)
     (at_kitchen_content content221)
     (not_allergic_gluten child314)
     (not_allergic_gluten child400)
     (waiting child314 table188)
     (waiting child400 table165)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child314)
     (served child400)
    )
  )
)

; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 643129

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child278 child25 - child
    bread224 bread318 - bread-portion
    content239 content271 - content-portion
    tray371 - tray
    table153 table387 table31 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_bread bread318)
     (at_kitchen_content content239)
     (at_kitchen_content content271)
     (no_gluten_bread bread224)
     (no_gluten_content content271)
     (allergic_gluten child278)
     (not_allergic_gluten child25)
     (waiting child278 table153)
     (waiting child25 table153)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child278)
     (served child25)
    )
  )
)

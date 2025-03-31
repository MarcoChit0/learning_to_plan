; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 977613

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child379 child369 - child
    bread48 bread430 - bread-portion
    content62 content32 - content-portion
    tray497 tray33 tray390 tray409 - tray
    table383 table186 table253 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray497 kitchen)
     (at tray33 kitchen)
     (at tray390 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread48)
     (at_kitchen_bread bread430)
     (at_kitchen_content content62)
     (at_kitchen_content content32)
     (not_allergic_gluten child369)
     (not_allergic_gluten child379)
     (waiting child379 table383)
     (waiting child369 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child379)
     (served child369)
    )
  )
)

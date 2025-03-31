; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 744156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child231 - child
    bread313 bread243 - bread-portion
    content449 content0 - content-portion
    tray410 tray258 tray82 - tray
    table177 table66 table39 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at tray258 kitchen)
     (at tray82 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_bread bread243)
     (at_kitchen_content content449)
     (at_kitchen_content content0)
     (no_gluten_bread bread313)
     (no_gluten_content content449)
     (allergic_gluten child12)
     (not_allergic_gluten child231)
     (waiting child12 table66)
     (waiting child231 table39)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child12)
     (served child231)
    )
  )
)

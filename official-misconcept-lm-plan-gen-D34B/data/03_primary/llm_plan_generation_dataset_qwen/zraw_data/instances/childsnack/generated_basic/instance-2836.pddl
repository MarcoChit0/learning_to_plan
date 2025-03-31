; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 38694

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child379 child231 - child
    bread352 bread359 - bread-portion
    content236 content300 - content-portion
    tray482 tray352 - tray
    table188 table414 table118 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray352 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread359)
     (at_kitchen_content content236)
     (at_kitchen_content content300)
     (not_allergic_gluten child231)
     (not_allergic_gluten child379)
     (waiting child379 table188)
     (waiting child231 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child379)
     (served child231)
    )
  )
)

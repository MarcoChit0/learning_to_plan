; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 782092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child222 - child
    bread52 bread207 - bread-portion
    content44 content111 - content-portion
    tray231 - tray
    table66 table455 table319 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at_kitchen_bread bread52)
     (at_kitchen_bread bread207)
     (at_kitchen_content content44)
     (at_kitchen_content content111)
     (no_gluten_bread bread52)
     (no_gluten_content content44)
     (allergic_gluten child442)
     (not_allergic_gluten child222)
     (waiting child442 table319)
     (waiting child222 table455)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child222)
    )
  )
)

; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 7878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child216 child215 - child
    bread498 bread119 - bread-portion
    content327 content250 - content-portion
    tray147 tray435 - tray
    table66 table217 table274 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray147 kitchen)
     (at tray435 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_bread bread119)
     (at_kitchen_content content327)
     (at_kitchen_content content250)
     (not_allergic_gluten child215)
     (not_allergic_gluten child216)
     (waiting child216 table274)
     (waiting child215 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child216)
     (served child215)
    )
  )
)

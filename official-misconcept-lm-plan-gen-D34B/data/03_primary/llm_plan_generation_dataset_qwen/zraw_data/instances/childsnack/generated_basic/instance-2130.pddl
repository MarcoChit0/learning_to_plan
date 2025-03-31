; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 412693

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 child10 - child
    bread241 bread32 - bread-portion
    content467 content7 - content-portion
    tray97 tray160 tray259 - tray
    table295 table411 table365 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at tray160 kitchen)
     (at tray259 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread32)
     (at_kitchen_content content467)
     (at_kitchen_content content7)
     (no_gluten_bread bread241)
     (no_gluten_content content467)
     (allergic_gluten child10)
     (not_allergic_gluten child427)
     (waiting child427 table411)
     (waiting child10 table411)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child427)
     (served child10)
    )
  )
)

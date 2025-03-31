; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 491335

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 child235 - child
    bread289 bread152 - bread-portion
    content113 content493 - content-portion
    tray25 tray261 - tray
    table303 table306 table411 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray25 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread289)
     (at_kitchen_bread bread152)
     (at_kitchen_content content113)
     (at_kitchen_content content493)
     (no_gluten_bread bread152)
     (no_gluten_content content113)
     (allergic_gluten child316)
     (not_allergic_gluten child235)
     (waiting child316 table411)
     (waiting child235 table306)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child316)
     (served child235)
    )
  )
)

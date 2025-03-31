; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 163681

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 child477 - child
    bread407 bread175 - bread-portion
    content263 content244 - content-portion
    tray407 tray439 - tray
    table315 table164 table487 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray407 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_bread bread175)
     (at_kitchen_content content263)
     (at_kitchen_content content244)
     (not_allergic_gluten child477)
     (not_allergic_gluten child215)
     (waiting child215 table315)
     (waiting child477 table487)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child215)
     (served child477)
    )
  )
)

; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 697857

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child200 - child
    bread294 bread107 - bread-portion
    content117 content53 - content-portion
    tray184 tray373 tray68 tray352 - tray
    table422 table347 table416 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray184 kitchen)
     (at tray373 kitchen)
     (at tray68 kitchen)
     (at tray352 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread107)
     (at_kitchen_content content117)
     (at_kitchen_content content53)
     (not_allergic_gluten child244)
     (not_allergic_gluten child200)
     (waiting child244 table347)
     (waiting child200 table422)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child244)
     (served child200)
    )
  )
)

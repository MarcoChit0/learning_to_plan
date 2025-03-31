; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 913100

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 child293 - child
    bread224 bread278 - bread-portion
    content289 content244 - content-portion
    tray316 tray362 - tray
    table139 table267 table183 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray316 kitchen)
     (at tray362 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_bread bread278)
     (at_kitchen_content content289)
     (at_kitchen_content content244)
     (not_allergic_gluten child293)
     (not_allergic_gluten child408)
     (waiting child408 table267)
     (waiting child293 table267)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child408)
     (served child293)
    )
  )
)

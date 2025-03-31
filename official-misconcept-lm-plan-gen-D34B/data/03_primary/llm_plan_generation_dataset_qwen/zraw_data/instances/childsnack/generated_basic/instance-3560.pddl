; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 200099

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child198 - child
    bread351 bread28 - bread-portion
    content397 content228 - content-portion
    tray259 tray221 - tray
    table83 table311 table97 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray259 kitchen)
     (at tray221 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread28)
     (at_kitchen_content content397)
     (at_kitchen_content content228)
     (not_allergic_gluten child198)
     (not_allergic_gluten child35)
     (waiting child35 table97)
     (waiting child198 table311)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child35)
     (served child198)
    )
  )
)

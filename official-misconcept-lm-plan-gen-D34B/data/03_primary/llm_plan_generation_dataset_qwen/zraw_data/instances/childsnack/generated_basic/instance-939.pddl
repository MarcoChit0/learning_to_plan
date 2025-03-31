; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 57959

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child397 child465 child101 - child
    bread351 bread139 bread308 - bread-portion
    content294 content35 content312 - content-portion
    tray91 tray134 tray421 tray16 - tray
    table459 table224 table453 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at tray134 kitchen)
     (at tray421 kitchen)
     (at tray16 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread308)
     (at_kitchen_content content294)
     (at_kitchen_content content35)
     (at_kitchen_content content312)
     (not_allergic_gluten child465)
     (not_allergic_gluten child397)
     (not_allergic_gluten child101)
     (waiting child397 table453)
     (waiting child465 table453)
     (waiting child101 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child397)
     (served child465)
     (served child101)
    )
  )
)

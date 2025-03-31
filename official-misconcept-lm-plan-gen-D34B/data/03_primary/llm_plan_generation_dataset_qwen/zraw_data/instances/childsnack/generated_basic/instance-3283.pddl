; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 607838

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 child167 - child
    bread356 bread312 - bread-portion
    content270 content341 - content-portion
    tray65 tray385 tray282 tray27 - tray
    table369 table470 table49 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at tray385 kitchen)
     (at tray282 kitchen)
     (at tray27 kitchen)
     (at_kitchen_bread bread356)
     (at_kitchen_bread bread312)
     (at_kitchen_content content270)
     (at_kitchen_content content341)
     (no_gluten_bread bread312)
     (no_gluten_content content270)
     (allergic_gluten child172)
     (not_allergic_gluten child167)
     (waiting child172 table49)
     (waiting child167 table369)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child172)
     (served child167)
    )
  )
)

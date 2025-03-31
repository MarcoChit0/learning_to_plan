; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 362803

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child384 child12 - child
    bread278 bread52 - bread-portion
    content293 content110 - content-portion
    tray453 tray270 - tray
    table377 table146 table41 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray453 kitchen)
     (at tray270 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread52)
     (at_kitchen_content content293)
     (at_kitchen_content content110)
     (not_allergic_gluten child12)
     (not_allergic_gluten child384)
     (waiting child384 table41)
     (waiting child12 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child384)
     (served child12)
    )
  )
)

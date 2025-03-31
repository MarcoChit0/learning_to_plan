; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 185383

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 child417 - child
    bread259 bread255 - bread-portion
    content59 content370 - content-portion
    tray185 tray18 tray229 tray96 - tray
    table80 table349 table88 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray18 kitchen)
     (at tray229 kitchen)
     (at tray96 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread255)
     (at_kitchen_content content59)
     (at_kitchen_content content370)
     (not_allergic_gluten child350)
     (not_allergic_gluten child417)
     (waiting child350 table349)
     (waiting child417 table80)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child350)
     (served child417)
    )
  )
)

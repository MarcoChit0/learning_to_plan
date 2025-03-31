; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 232108

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 child465 - child
    bread389 bread191 - bread-portion
    content96 content338 - content-portion
    tray490 tray185 - tray
    table65 table108 table344 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at tray185 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_bread bread191)
     (at_kitchen_content content96)
     (at_kitchen_content content338)
     (not_allergic_gluten child289)
     (not_allergic_gluten child465)
     (waiting child289 table65)
     (waiting child465 table344)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child289)
     (served child465)
    )
  )
)

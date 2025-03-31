; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 138575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child145 - child
    bread263 bread408 - bread-portion
    content495 content167 - content-portion
    tray206 tray35 tray230 - tray
    table293 table346 table55 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray35 kitchen)
     (at tray230 kitchen)
     (at_kitchen_bread bread263)
     (at_kitchen_bread bread408)
     (at_kitchen_content content495)
     (at_kitchen_content content167)
     (not_allergic_gluten child145)
     (not_allergic_gluten child338)
     (waiting child338 table346)
     (waiting child145 table346)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child338)
     (served child145)
    )
  )
)

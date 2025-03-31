; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 519233

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 - child
    bread483 - bread-portion
    content424 - content-portion
    tray388 tray89 - tray
    table41 table81 table276 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray388 kitchen)
     (at tray89 kitchen)
     (at_kitchen_bread bread483)
     (at_kitchen_content content424)
     (not_allergic_gluten child36)
     (waiting child36 table276)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child36)
    )
  )
)

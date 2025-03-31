; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 236042

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 - child
    bread292 - bread-portion
    content86 - content-portion
    tray293 tray116 - tray
    table49 table399 table242 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at tray116 kitchen)
     (at_kitchen_bread bread292)
     (at_kitchen_content content86)
     (not_allergic_gluten child389)
     (waiting child389 table399)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child389)
    )
  )
)

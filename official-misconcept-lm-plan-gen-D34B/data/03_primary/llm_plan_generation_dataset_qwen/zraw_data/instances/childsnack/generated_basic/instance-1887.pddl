; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 206548

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child412 child212 - child
    bread352 bread74 - bread-portion
    content89 content403 - content-portion
    tray68 tray498 - tray
    table104 table357 table41 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread74)
     (at_kitchen_content content89)
     (at_kitchen_content content403)
     (no_gluten_bread bread352)
     (no_gluten_content content403)
     (allergic_gluten child212)
     (not_allergic_gluten child412)
     (waiting child412 table357)
     (waiting child212 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child412)
     (served child212)
    )
  )
)

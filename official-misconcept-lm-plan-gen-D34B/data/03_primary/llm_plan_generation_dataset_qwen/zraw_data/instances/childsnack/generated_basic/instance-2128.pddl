; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 735940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child401 child396 - child
    bread115 bread24 - bread-portion
    content183 content294 - content-portion
    tray209 tray469 tray177 - tray
    table133 table6 table116 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray469 kitchen)
     (at tray177 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread24)
     (at_kitchen_content content183)
     (at_kitchen_content content294)
     (no_gluten_bread bread115)
     (no_gluten_content content183)
     (allergic_gluten child396)
     (not_allergic_gluten child401)
     (waiting child401 table116)
     (waiting child396 table6)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child401)
     (served child396)
    )
  )
)

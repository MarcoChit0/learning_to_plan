; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 409360

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 child264 - child
    bread192 bread131 - bread-portion
    content214 content191 - content-portion
    tray400 tray5 - tray
    table401 table221 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray400 kitchen)
     (at tray5 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread131)
     (at_kitchen_content content214)
     (at_kitchen_content content191)
     (no_gluten_bread bread192)
     (no_gluten_content content214)
     (allergic_gluten child264)
     (not_allergic_gluten child116)
     (waiting child116 table243)
     (waiting child264 table243)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child116)
     (served child264)
    )
  )
)

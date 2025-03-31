; child-snack task with 3 children and 0.74 gluten factor 
; constant factor of 1.3
; random seed: 935945

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child86 child43 child483 - child
    bread52 bread66 bread265 - bread-portion
    content427 content334 content262 - content-portion
    tray293 - tray
    table264 table203 table61 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at_kitchen_bread bread52)
     (at_kitchen_bread bread66)
     (at_kitchen_bread bread265)
     (at_kitchen_content content427)
     (at_kitchen_content content334)
     (at_kitchen_content content262)
     (no_gluten_bread bread52)
     (no_gluten_bread bread265)
     (no_gluten_content content334)
     (no_gluten_content content427)
     (allergic_gluten child43)
     (allergic_gluten child86)
     (not_allergic_gluten child483)
     (waiting child86 table203)
     (waiting child43 table264)
     (waiting child483 table264)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child86)
     (served child43)
     (served child483)
    )
  )
)

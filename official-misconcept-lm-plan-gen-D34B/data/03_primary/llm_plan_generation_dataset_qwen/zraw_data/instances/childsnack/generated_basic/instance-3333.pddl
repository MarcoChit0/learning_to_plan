; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 565960

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child182 child27 - child
    bread341 bread492 - bread-portion
    content371 content265 - content-portion
    tray344 tray324 tray105 tray364 - tray
    table11 table264 table206 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray324 kitchen)
     (at tray105 kitchen)
     (at tray364 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_bread bread492)
     (at_kitchen_content content371)
     (at_kitchen_content content265)
     (no_gluten_bread bread341)
     (no_gluten_content content371)
     (allergic_gluten child182)
     (not_allergic_gluten child27)
     (waiting child182 table11)
     (waiting child27 table264)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child182)
     (served child27)
    )
  )
)
